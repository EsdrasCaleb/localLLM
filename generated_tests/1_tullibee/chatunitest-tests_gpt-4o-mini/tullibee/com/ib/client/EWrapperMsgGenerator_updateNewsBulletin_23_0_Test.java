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
    public void testUpdateNewsBulletin_ValidInputs() {
        // Arrange
        int msgId = 1;
        int msgType = 2;
        String message = "This is a test message";
        String origExchange = "NYSE";
        // Act
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assert
        assertEquals("MsgId=1 :: MsgType=2 :: Origin=NYSE :: Message=This is a test message", result);
    }

    @Test
    public void testUpdateNewsBulletin_EmptyMessage() {
        // Arrange
        int msgId = 1;
        int msgType = 2;
        String message = "";
        String origExchange = "NASDAQ";
        // Act
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assert
        assertEquals("MsgId=1 :: MsgType=2 :: Origin=NASDAQ :: Message=", result);
    }

    @Test
    public void testUpdateNewsBulletin_NullMessage() {
        // Arrange
        int msgId = 1;
        int msgType = 2;
        String message = null;
        String origExchange = "LSE";
        // Act
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assert
        assertEquals("MsgId=1 :: MsgType=2 :: Origin=LSE :: Message=null", result);
    }

    @Test
    public void testUpdateNewsBulletin_NegativeMsgId() {
        // Arrange
        int msgId = -1;
        int msgType = 2;
        String message = "Negative ID test";
        String origExchange = "CBOE";
        // Act
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assert
        assertEquals("MsgId=-1 :: MsgType=2 :: Origin=CBOE :: Message=Negative ID test", result);
    }

    @Test
    public void testUpdateNewsBulletin_NegativeMsgType() {
        // Arrange
        int msgId = 1;
        int msgType = -2;
        String message = "Negative Type test";
        String origExchange = "TSX";
        // Act
        String result = EWrapperMsgGenerator.updateNewsBulletin(msgId, msgType, message, origExchange);
        // Assert
        assertEquals("MsgId=1 :: MsgType=-2 :: Origin=TSX :: Message=Negative Type test", result);
    }
}
