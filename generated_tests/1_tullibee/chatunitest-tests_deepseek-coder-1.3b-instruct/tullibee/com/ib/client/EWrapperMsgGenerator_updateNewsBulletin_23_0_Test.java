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
        String result = EWrapperMsgGenerator.updateNewsBulletin(1, 2, "Test Message", "Test Exchange");
        assertEquals("MsgId=1 :: MsgType=2 :: Origin=Test Exchange :: Message=Test Message", result);
        result = EWrapperMsgGenerator.updateNewsBulletin(3, 4, "Another Test Message", "Another Test Exchange");
        assertEquals("MsgId=3 :: MsgType=4 :: Origin=Another Test Exchange :: Message=Another Test Message", result);
        result = EWrapperMsgGenerator.updateNewsBulletin(5, 6, "Message with spaces", "Spaces in the message");
        assertEquals("MsgId=5 :: MsgType=6 :: Origin=Spaces in the message :: Message=Message with spaces", result);
        // Add more test cases as needed
    }
}
