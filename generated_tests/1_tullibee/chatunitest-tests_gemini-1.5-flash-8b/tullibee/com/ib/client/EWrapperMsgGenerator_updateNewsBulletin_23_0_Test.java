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
    void testUpdateNewsBulletin_ValidInput() {
        int msgId = 123;
        int msgType = 456;
        String message = "This is a news bulletin.";
        String origExchange = "NASDAQ";
        String expectedOutput = "MsgId=123 :: MsgType=456 :: Origin=NASDAQ :: Message=This is a news bulletin.";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateNewsBulletin_ZeroMsgId() {
        int msgId = 0;
        int msgType = 456;
        String message = "Another news bulletin.";
        String origExchange = "NYSE";
        String expectedOutput = "MsgId=0 :: MsgType=456 :: Origin=NYSE :: Message=Another news bulletin.";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateNewsBulletin_NegativeMsgType() {
        int msgId = 123;
        int msgType = -456;
        String message = "A different news bulletin.";
        String origExchange = "ARCA";
        String expectedOutput = "MsgId=123 :: MsgType=-456 :: Origin=ARCA :: Message=A different news bulletin.";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateNewsBulletin_NullMessage() {
        int msgId = 123;
        int msgType = 456;
        String message = null;
        String origExchange = "SMART";
        String expectedOutput = "MsgId=123 :: MsgType=456 :: Origin=SMART :: Message=null";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateNewsBulletin_EmptyMessage() {
        int msgId = 123;
        int msgType = 456;
        String message = "";
        String origExchange = "SMART";
        String expectedOutput = "MsgId=123 :: MsgType=456 :: Origin=SMART :: Message=";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateNewsBulletin_NullOrigExchange() {
        int msgId = 123;
        int msgType = 456;
        String message = "News bulletin.";
        String origExchange = null;
        String expectedOutput = "MsgId=123 :: MsgType=456 :: Origin=null :: Message=News bulletin.";
        String actualOutput = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        assertEquals(expectedOutput, actualOutput);
    }
}
