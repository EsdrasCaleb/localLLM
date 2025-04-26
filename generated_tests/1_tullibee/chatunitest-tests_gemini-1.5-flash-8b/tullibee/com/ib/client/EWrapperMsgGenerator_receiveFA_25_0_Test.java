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

class EWrapperMsgGenerator_receiveFA_25_0_Test {

    @Test
    void testReceiveFA_ValidInput() {
        String xmlData = "<xmlData />";
        int dataType = 1;
        String expectedOutput = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(dataType) + " " + xmlData;
        String actualOutput = EWrapperMsgGenerator.receiveFA(dataType, xmlData);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testReceiveFA_NullXML() {
        int dataType = 2;
        String xmlData = null;
        String expectedOutput = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(dataType) + " " + xmlData;
        String actualOutput = EWrapperMsgGenerator.receiveFA(dataType, xmlData);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testReceiveFA_EmptyXML() {
        int dataType = 3;
        String xmlData = "";
        String expectedOutput = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(dataType) + " " + xmlData;
        String actualOutput = EWrapperMsgGenerator.receiveFA(dataType, xmlData);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testReceiveFA_ZeroDataType() {
        String xmlData = "<xmlData />";
        int dataType = 0;
        String expectedOutput = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(dataType) + " " + xmlData;
        String actualOutput = EWrapperMsgGenerator.receiveFA(dataType, xmlData);
        assertEquals(expectedOutput, actualOutput);
    }

    // Additional test for edge case.
    @Test
    void testReceiveFA_LargeDataType() {
        String xmlData = "<xmlData />";
        int dataType = Integer.MAX_VALUE;
        String expectedOutput = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(dataType) + " " + xmlData;
        String actualOutput = EWrapperMsgGenerator.receiveFA(dataType, xmlData);
        assertEquals(expectedOutput, actualOutput);
    }
}

// Dummy class to avoid compilation errors (replace with your actual EClientSocket class)
class EClientSocket {

    static String faMsgTypeName(int faDataType) {
        switch(faDataType) {
            case 0:
                return "TYPE_0";
            case 1:
                return "TYPE_1";
            case 2:
                return "TYPE_2";
            case 3:
                return "TYPE_3";
            case 4:
                return "MOCKED_TYPE_NAME";
            default:
                return "UNKNOWN_TYPE_" + faDataType;
        }
    }
}
