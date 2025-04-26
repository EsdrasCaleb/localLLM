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

class EWrapperMsgGenerator_tickOptionComputation_2_3_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    void testTickOptionComputationValidInputs() {
        int tickerId = 12345;
        int field = 1;
        double impliedVol = 0.5;
        double delta = 0.1;
        double modelPrice = 20.0;
        double pvDividend = 10.0;
        String expectedOutput = "id=12345  Call: vol = 0.5 delta = 0.1 modelPrice = 20.0 pvDividend = 10.0";
        String actualOutput = eWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickOptionComputationInvalidImpliedVol() {
        int tickerId = 12345;
        int field = 1;
        // Invalid input
        double impliedVol = -1.0;
        double delta = 0.1;
        double modelPrice = 20.0;
        double pvDividend = 10.0;
        String expectedOutput = "id=12345  Call: vol = N/A delta = 0.1 modelPrice = 20.0 pvDividend = 10.0";
        String actualOutput = eWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickOptionComputationInvalidDelta() {
        int tickerId = 12345;
        int field = 1;
        double impliedVol = 0.5;
        // Invalid input
        double delta = -0.1;
        double modelPrice = 20.0;
        double pvDividend = 10.0;
        String expectedOutput = "id=12345  Call: vol = 0.5 delta = N/A modelPrice = 20.0 pvDividend = 10.0";
        String actualOutput = eWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickOptionComputationInvalidModelPrice() {
        int tickerId = 12345;
        int field = 1;
        double impliedVol = 0.5;
        double delta = 0.1;
        // Invalid input
        double modelPrice = -20.0;
        double pvDividend = 10.0;
        String expectedOutput = "id=12345  Call: vol = 0.5 delta = 0.1 modelPrice = N/A pvDividend = 10.0";
        String actualOutput = eWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testTickOptionComputationInvalidPvDividend() {
        int tickerId = 12345;
        int field = 1;
        double impliedVol = 0.5;
        double delta = 0.1;
        double modelPrice = 20.0;
        // Invalid input
        double pvDividend = -10.0;
        String expectedOutput = "id=12345  Call: vol = 0.5 delta = 0.1 modelPrice = 20.0 pvDividend = N/A";
        String actualOutput = eWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedOutput, actualOutput);
    }
}
