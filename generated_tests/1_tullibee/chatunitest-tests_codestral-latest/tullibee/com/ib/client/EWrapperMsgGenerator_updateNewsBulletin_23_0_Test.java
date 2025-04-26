package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_updateNewsBulletin_23_0_Test {

    @Test
    public void testUpdateNewsBulletin() {
        int msgId = 1;
        int msgType = 2;
        String message = "Test message";
        String origExchange = "Test exchange";
        String expected = "MsgId=1 :: MsgType=2 :: Origin=Test exchange :: Message=Test message";
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expected, result);
    }
}
