package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.DataInputStream;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EReader_run_1_0_Test {

    @Mock
    private DataInputStream m_dis;

    @Mock
    private EWrapper m_eWrapper;

    @Mock
    private int m_serverVersion;

    @InjectMocks
    private EReader eReader;

    @Test
    public void testRun() throws Exception {
        // Arrange
        when(m_dis.readInt()).thenReturn(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 4, 5);
        // Act
        eReader.run();
        // Assert
        verify(m_eWrapper, never()).stopRequested();
        verify(m_eWrapper, never()).connectionClosed();
        verify(m_eWrapper, never()).error(any(Exception.class));
    }
}
