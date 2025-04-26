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
        // Test data
        int msgId = 123;
        int msgType = 456;
        String message = "Test News Bulletin";
        String origExchange = "Test Exchange";
        // Expected result
        String expected = "MsgId=123 :: MsgType=456 :: Origin=Test Exchange :: Message=Test News Bulletin";
        // Actual result from the method under test
        String actual = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assertion
        assertEquals(expected, actual, "The formatted news bulletin string does not match the expected output.");
    }
}
