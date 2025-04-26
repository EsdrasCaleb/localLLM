package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;

class EWrapperMsgGenerator_updateNewsBulletin_23_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class EWrapperMsgGeneratorUpdateNewsBulletinTest {

        @Test
        public void testUpdateNewsBulletin() {
            // Setup
            int msgId = 123;
            int msgType = 456;
            String message = "New Message";
            String origExchange = "ABC";
            // Mocking
            String expectedMessage = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
            when(EWrapperMsgGenerator.updateNewsBulletin(anyInt(), anyInt(), anyString(), anyString())).thenReturn(expectedMessage);
            // Execution
            String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
            // Verification
            assertEquals(expectedMessage, result);
        }
    }
}
