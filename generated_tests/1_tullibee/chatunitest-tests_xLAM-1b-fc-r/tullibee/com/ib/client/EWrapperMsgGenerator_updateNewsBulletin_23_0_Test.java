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

class EWrapperMsgGenerator_updateNewsBulletin_23_0_Test {

    @Test
    void updateNewsBulletinTest() {
        // Given
        int msgId = 123;
        int msgType = 456;
        String message = "This is a test message.";
        String origExchange = "Test exchange";
        // When
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Then
        String expected = "MsgId=" + msgId + " :: MsgType=" + msgType + " :: Origin=" + origExchange + " :: Message=" + message;
        assertEquals(expected, result);
    }
}
