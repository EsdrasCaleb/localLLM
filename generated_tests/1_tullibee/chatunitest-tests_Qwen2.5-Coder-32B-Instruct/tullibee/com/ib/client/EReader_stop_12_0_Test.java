package com.ib.client;

import java.io.DataInputStream;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EReader_stop_12_0_Test {

    private EReader eReader;

    private DataInputStream mockDataInputStream;

    private EWrapper mockEWrapper;

    @BeforeEach
    public void setUp() throws Exception {
        mockDataInputStream = mock(DataInputStream.class);
        mockEWrapper = mock(EWrapper.class);
        eReader = new EReader(mockDataInputStream, mockEWrapper, 1);
    }

    @Test
    public void testStop() throws Exception {
        // Initially, m_stopRequested should be false
        assertFalse(getStopRequestedField());
        // Call the stop method
        eReader.stop();
        // After calling stop, m_stopRequested should be true
        assertTrue(getStopRequestedField());
    }

    private boolean getStopRequestedField() throws Exception {
        Field stopRequestedField = EReader.class.getDeclaredField("m_stopRequested");
        stopRequestedField.setAccessible(true);
        return (boolean) stopRequestedField.get(eReader);
    }
}
