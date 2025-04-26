package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UNKNOWN_ID;
import java.io.DataInputStream;
import java.io.IOException;
import java.util.Vector;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class EReader_run_1_1_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestRunner {

        @Test
        public void testRun() throws Exception {
            // Arrange
            DataInputStream dis = mock(DataInputStream.class);
            EWrapper eWrapper = mock(EWrapper.class);
            int serverVersion = 10;
            EReader reader = new EReader(dis, eWrapper, serverVersion);
            // Act
            reader.run();
            // Assert
            verify(eWrapper).stopRequested();
            verify(eWrapper).connectionClosed();
        }
    }
}
