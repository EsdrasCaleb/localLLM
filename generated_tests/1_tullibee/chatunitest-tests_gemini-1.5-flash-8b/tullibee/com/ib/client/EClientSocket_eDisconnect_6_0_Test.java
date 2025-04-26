package com.ib.client;

import java.io.IOException;
import java.net.Socket;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import com.ib.client.EClientErrors.CodeMsgPair;
// Import the class
import com.ib.client.EClientSocket;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class EClientSocket_eDisconnect_6_0_Test {

    @Test
    void eDisconnect_socketNotNull_closesSocket() throws IOException {
        // Mock the socket
        Socket mockSocket = Mockito.mock(Socket.class);
        // Crucial:  Do nothing for close() if socket is null
        doNothing().when(mockSocket).close();
        // Create an EClientSocket instance
        EClientSocket socket = new EClientSocket();
        // Set the socket field using reflection
        try {
            java.lang.reflect.Field socketField = EClientSocket.class.getDeclaredField("m_socket");
            socketField.setAccessible(true);
            socketField.set(socket, mockSocket);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing m_socket field: " + e.getMessage());
        }
        // Call the method under test
        socket.eDisconnect();
        // Verify that the socket was closed
        verify(mockSocket).close();
    }

    @Test
    void eDisconnect_socketNull_doesNothing() throws IOException {
        EClientSocket socket = new EClientSocket();
        // Crucial:  Set socket to null
        try {
            java.lang.reflect.Field socketField = EClientSocket.class.getDeclaredField("m_socket");
            socketField.setAccessible(true);
            socketField.set(socket, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing m_socket field: " + e.getMessage());
        }
        socket.eDisconnect();
        // No need to verify anything specific since no action should be taken
    }
}
