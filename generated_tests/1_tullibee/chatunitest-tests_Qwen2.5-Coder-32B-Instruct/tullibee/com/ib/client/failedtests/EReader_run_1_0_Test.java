package com.ib.client;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
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

    private EReader eReader;

    private DataInputStream m_dis;

    private EWrapper m_eWrapper;

    private int m_serverVersion;

    private volatile boolean m_stopRequested;

    @BeforeEach
    public void setUp() throws IOException {
        m_serverVersion = 1;
        m_eWrapper = mock(EWrapper.class);
        m_dis = new DataInputStream(new ByteArrayInputStream(new byte[0]));
        eReader = new EReader(m_dis, m_eWrapper, m_serverVersion);
    }

    @Test
    public void testRunStopRequested() throws IOException, NoSuchFieldException, IllegalAccessException {
        // Set m_stopRequested to true
        setStopRequested(true);
        // Run the method
        eReader.run();
        // Verify that stopRequested was called on m_eWrapper
        verify(m_eWrapper, times(1)).stopRequested();
        verify(m_eWrapper, times(0)).connectionClosed();
        verify(m_eWrapper, times(0)).error(any(Exception.class));
    }

    @Test
    public void testRunContinueProcessingFalse() throws IOException, NoSuchFieldException, IllegalAccessException {
        // Mock processMsg to return false
        mockProcessMsg(false);
        // Run the method
        eReader.run();
        // Verify that connectionClosed was called on m_eWrapper
        verify(m_eWrapper, times(0)).stopRequested();
        verify(m_eWrapper, times(1)).connectionClosed();
        verify(m_eWrapper, times(0)).error(any(Exception.class));
    }

    @Test
    public void testRunException() throws IOException, NoSuchFieldException, IllegalAccessException {
        // Mock processMsg to throw an exception
        mockProcessMsgException(new IOException("Test Exception"));
        // Run the method
        eReader.run();
        // Verify that error was called on m_eWrapper
        verify(m_eWrapper, times(0)).stopRequested();
        verify(m_eWrapper, times(0)).connectionClosed();
        verify(m_eWrapper, times(1)).error(any(Exception.class));
    }

    private void setStopRequested(boolean value) throws NoSuchFieldException, IllegalAccessException {
        var field = EReader.class.getDeclaredField("m_stopRequested");
        field.setAccessible(true);
        field.set(eReader, value);
    }

    private void mockProcessMsg(boolean value) throws IOException {
        try (var dis = Mockito.spy(new DataInputStream(new ByteArrayInputStream(new byte[4])))) {
            // Assuming 1 is a valid message type for testing
            when(dis.readInt()).thenReturn(1);
            eReader = new EReader(dis, m_eWrapper, m_serverVersion);
            var processMsgMethod = EReader.class.getDeclaredMethod("processMsg", int.class);
            processMsgMethod.setAccessible(true);
            when(processMsgMethod.invoke(eReader, anyInt())).thenReturn(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void mockProcessMsgException(Exception ex) throws IOException {
        try (var dis = Mockito.spy(new DataInputStream(new ByteArrayInputStream(new byte[4])))) {
            when(dis.readInt()).thenThrow(ex);
            eReader = new EReader(dis, m_eWrapper, m_serverVersion);
        }
    }
}
