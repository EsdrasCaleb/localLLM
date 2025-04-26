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

class EWrapperMsgGenerator_updateAccountTime_11_0_Test {

    @Test
    void testUpdateAccountTime_validInput() {
        String timeStamp = "2023-10-27 10:00:00";
        String expectedOutput = "updateAccountTime: 2023-10-27 10:00:00";
        String actualOutput = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountTime_nullInput() {
        String timeStamp = null;
        String expectedOutput = "updateAccountTime: null";
        String actualOutput = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountTime_emptyInput() {
        String timeStamp = "";
        String expectedOutput = "updateAccountTime: ";
        String actualOutput = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountTime_longInput() {
        String timeStamp = "2024-01-15 23:59:59";
        String expectedOutput = "updateAccountTime: 2024-01-15 23:59:59";
        String actualOutput = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expectedOutput, actualOutput);
    }
}
