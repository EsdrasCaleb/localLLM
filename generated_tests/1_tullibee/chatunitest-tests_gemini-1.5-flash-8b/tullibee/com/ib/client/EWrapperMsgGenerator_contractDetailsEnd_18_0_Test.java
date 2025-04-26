package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_contractDetailsEnd_18_0_Test {

    @Test
    void testContractDetailsEnd() throws Exception {
        // Test case 1: Valid request ID
        int reqId = 123;
        String expectedOutput = "123 =============== end ===============";
        String actualOutput = EWrapperMsgGenerator.contractDetailsEnd(reqId);
        assertEquals(expectedOutput, actualOutput);
        // Test case 2: Zero request ID
        reqId = 0;
        expectedOutput = "0 =============== end ===============";
        actualOutput = EWrapperMsgGenerator.contractDetailsEnd(reqId);
        assertEquals(expectedOutput, actualOutput);
        // Test case 3: Negative request ID
        reqId = -1;
        expectedOutput = "-1 =============== end ===============";
        actualOutput = EWrapperMsgGenerator.contractDetailsEnd(reqId);
        assertEquals(expectedOutput, actualOutput);
    }
}
