package com.ib.client;

import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EClientSocket_eConnect_4_0_Test {

    @Test
    public void testEConnect() throws IOException {
        // Given
        EClientSocket eClientSocket = new EClientSocket();
        String host = "localhost";
        int port = 1234;
        int clientId = 5678;
        // When
        Exception exception = assertThrows(IOException.class, () -> {
            eClientSocket.eConnect(host, port, clientId);
        });
        // Then
        String expectedMessage = "No valid ID, already connected";
        String actualMessage = exception.getMessage();
        assert (actualMessage.contains(expectedMessage));
    }
}
