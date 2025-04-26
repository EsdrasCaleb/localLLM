package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EReader_run_1_0_Test {

    @Mock
    private DataInputStream dis;

    @Mock
    private EWrapper eWrapper;

    @InjectMocks
    private EReader eReader;

    @Test
    public void runTest() throws IOException {
        // Given
        int serverVersion = 123;
        doNothing().when(eReader).run();
        // When
        eReader.run();
        // Then
        verify(eWrapper, times(1)).stopRequested();
        verify(eWrapper, times(1)).connectionClosed();
        verify(eWrapper, times(1)).error(any(Exception.class));
    }
}
