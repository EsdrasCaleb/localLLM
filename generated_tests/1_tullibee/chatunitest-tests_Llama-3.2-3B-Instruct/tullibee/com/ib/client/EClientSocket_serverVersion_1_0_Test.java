package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_serverVersion_1_0_Test {

    @Mock
    private EClientSocket focal;

    @Test
    public void testServerVersion() {
        // Arrange
        when(focal.isConnected()).thenReturn(true);
        // Act
        try {
            int serverVersion = focal.serverVersion();
            assertEquals(46, serverVersion);
        } catch (Exception e) {
            // Expected exception
        }
        // Arrange
        when(focal.isConnected()).thenReturn(false);
        // Act
        try {
            int serverVersion = focal.serverVersion();
            assertEquals(46, serverVersion);
        } catch (Exception e) {
            // Expected exception
        }
    }

    @Test
    public void testServerVersionWithException() {
        // Arrange
        when(focal.isConnected()).thenReturn(false);
        // Act and Assert
        try {
            focal.serverVersion();
            assert false;
        } catch (EException e) {
            // Expected exception
        }
    }
}
