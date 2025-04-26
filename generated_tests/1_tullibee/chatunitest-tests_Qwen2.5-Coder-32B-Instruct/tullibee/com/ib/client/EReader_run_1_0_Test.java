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

    private void setStopRequested(boolean value) throws NoSuchFieldException, IllegalAccessException {
        var field = EReader.class.getDeclaredField("m_stopRequested");
        field.setAccessible(true);
        field.set(eReader, value);
    }
}
