package com.ib.client;

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
import java.io.IOException;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EClientSocket_dataInputStream_51_0_Test {

    @Test
    void test_dataInputStream() throws Exception {
        EClientSocket eClientSocket = Mockito.mock(EClientSocket.class);
        Mockito.when(eClientSocket.dataInputStream()).thenReturn(Mockito.mock(DataInputStream.class));
        DataInputStream dataInputStream = eClientSocket.dataInputStream();
        assertTrue(dataInputStream != null);
    }
}
