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

class EWrapperMsgGenerator_fundamentalData_32_0_Test {

    @Test
    void fundamentalData_validInput_returnsCorrectString() {
        int reqId = 123;
        String data = "some fundamental data";
        String expected = "id  = 123 len = 20\n" + data;
        String actual = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, actual);
    }

    @Test
    void fundamentalData_emptyData_returnsCorrectString() {
        int reqId = 456;
        String data = "";
        String expected = "id  = 456 len = 0\n";
        String actual = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, actual);
    }

    @Test
    void fundamentalData_nullData_returnsCorrectString() {
        int reqId = 789;
        String data = null;
        String expected = "id  = 789 len = 0\n";
        // Crucial:  Handle potential NullPointerException
        String actual = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, actual);
    }

    @Test
    void fundamentalData_largeData_returnsCorrectString() {
        int reqId = 10;
        String data = "This is a very long string that will test the length calculation.";
        String expected = "id  = 10 len = 67\n" + data;
        String actual = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, actual);
    }
}
