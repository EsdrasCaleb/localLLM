package com.ib.client;

import com.ib.client.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
class EReader_run_1_0_Test {

    @Mock
    private DataInputStream mockDis;

    @Mock
    private EWrapper mockEWrapper;

    @InjectMocks
    private EReader eReader;

    private int mockServerVersion = 1;

    @BeforeEach
    void setUp() {
        eReader = new EReader(mockDis, mockEWrapper, mockServerVersion);
    }

    @Test
    void testRun_stopRequested() throws Exception {
        when(mockDis.readInt()).thenReturn(EReader.NEXT_VALID_ID);
        // Correctly set stopRequested using reflection.  Crucially, do this *before* calling run().
        java.lang.reflect.Field stopRequestedField = EReader.class.getDeclaredField("m_stopRequested");
        stopRequestedField.setAccessible(true);
        stopRequestedField.set(eReader, true);
        doNothing().when(mockEWrapper).stopRequested();
        eReader.run();
        verify(mockEWrapper).stopRequested();
        verify(mockDis, times(1)).readInt();
    }

    @Test
    void testRun_connectionClosed() throws Exception {
        when(mockDis.readInt()).thenReturn(EReader.NEXT_VALID_ID).thenReturn(-1);
        doNothing().when(mockEWrapper).connectionClosed();
        eReader.run();
        verify(mockEWrapper).connectionClosed();
        verify(mockDis, times(2)).readInt();
    }

    @Test
    void testRun_error() throws Exception {
        when(mockDis.readInt()).thenReturn(EReader.NEXT_VALID_ID);
        RuntimeException mockException = new RuntimeException("Test Exception");
        doNothing().when(mockEWrapper).error(mockException);
        eReader.run();
        verify(mockEWrapper).error(mockException);
        verify(mockDis, times(1)).readInt();
    }

    @Test
    void testRun_normalFlow() throws Exception {
        when(mockDis.readInt()).thenReturn(EReader.NEXT_VALID_ID).thenReturn(EReader.NEXT_VALID_ID);
        eReader.run();
        verify(mockDis, times(2)).readInt();
        verifyNoInteractions(mockEWrapper);
    }
}
