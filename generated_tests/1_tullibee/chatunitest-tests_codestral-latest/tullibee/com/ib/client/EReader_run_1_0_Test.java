package com.ib.client;

import java.io.DataInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EReader_run_1_0_Test {

    @Mock
    private DataInputStream m_dis;

    @Mock
    private EWrapper m_eWrapper;

    @InjectMocks
    private EReader eReader;

    @BeforeEach
    public void setUp() throws Exception {
        Field stopRequestedField = EReader.class.getDeclaredField("m_stopRequested");
        stopRequestedField.setAccessible(true);
        stopRequestedField.set(eReader, false);
    }

    @Test
    public void testRunStopRequested() throws IOException, NoSuchFieldException, IllegalAccessException, EException {
        when(m_dis.readInt()).thenReturn(1);
        when(eReader.processMsg(1)).thenReturn(true);
        Field stopRequestedField = EReader.class.getDeclaredField("m_stopRequested");
        stopRequestedField.setAccessible(true);
        stopRequestedField.set(eReader, true);
        // Act
        eReader.run();
        // Assert
        verify(m_eWrapper).stopRequested();
        verify(m_eWrapper, never()).connectionClosed();
        verify(m_eWrapper, never()).error(any(Exception.class));
    }

    @Test
    public void testRunConnectionClosed() throws IOException, EException {
        when(m_dis.readInt()).thenReturn(1);
        when(eReader.processMsg(1)).thenReturn(false);
        // Act
        eReader.run();
        // Assert
        verify(m_eWrapper).connectionClosed();
        verify(m_eWrapper, never()).stopRequested();
        verify(m_eWrapper, never()).error(any(Exception.class));
    }

    @Test
    public void testRunException() throws IOException, EException {
        // Arrange
        when(m_dis.readInt()).thenThrow(new IOException("Test Exception"));
        // Act
        eReader.run();
        // Assert
        verify(m_eWrapper).error(any(Exception.class));
        verify(m_eWrapper, never()).stopRequested();
        verify(m_eWrapper, never()).connectionClosed();
    }
}
