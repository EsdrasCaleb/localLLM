package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_scannerDataEnd_30_0_Test {

    @ParameterizedTest
    @CsvSource({ "1, id = 1 =============== end ===============", "10, id = 10 =============== end ===============", "25, id = 25 =============== end ===============" })
    void scannerDataEnd_shouldReturnCorrectString(int reqId, String expectedOutput) {
        String actualOutput = EWrapperMsgGenerator.scannerDataEnd(reqId);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void scannerDataEnd_withZeroReqId() {
        String actualOutput = EWrapperMsgGenerator.scannerDataEnd(0);
        String expectedOutput = "id = 0 =============== end ===============";
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void scannerDataEnd_withNegativeReqId() {
        String actualOutput = EWrapperMsgGenerator.scannerDataEnd(-1);
        String expectedOutput = "id = -1 =============== end ===============";
        assertEquals(expectedOutput, actualOutput);
    }
}
