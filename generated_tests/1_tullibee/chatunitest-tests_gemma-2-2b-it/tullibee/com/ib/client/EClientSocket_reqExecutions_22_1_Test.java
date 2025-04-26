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

public class EClientSocket_reqExecutions_22_1_Test {

    @Test
    void testReqExecutions() {
        EClientSocket clientSocket = new EClientSocket();
        ExecutionFilter filter = new ExecutionFilter();
        filter.m_clientId = 123;
        filter.m_acctCode = "ABC";
        filter.m_time = "2023-10-27-10:00:00";
        filter.m_symbol = "AAPL";
        filter.m_secType = "STK";
        filter.m_exchange = "NYSE";
        filter.m_side = "BUY";
        try {
            clientSocket.reqExecutions(1, filter);
        } catch (IOException | EException e) {
            fail(e.getMessage());
        }
    }
}
