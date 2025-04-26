package com.ib.client;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
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
class EClientSocket_cancelScannerSubscription_7_4_Test {

    @Test
    void cancelScannerSubscriptionTest(MockedStatic<EClientSocket> mockedSocket) throws Exception {
        // Given
        EClientSocket eClientSocket = new EClientSocket();
        int tickerId = 123;
        // When
        eClientSocket.cancelScannerSubscription(tickerId);
        // Then
        verify(eClientSocket, times(1)).send(anyInt());
        verify(eClientSocket, times(1)).send(anyInt());
        verify(eClientSocket, times(1)).send(anyInt());
    }
}
