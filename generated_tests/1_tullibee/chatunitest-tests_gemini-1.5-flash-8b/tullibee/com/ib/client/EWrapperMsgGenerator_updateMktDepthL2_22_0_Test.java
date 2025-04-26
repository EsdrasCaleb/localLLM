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

class EWrapperMsgGenerator_updateMktDepthL2_22_0_Test {

    @Test
    void testUpdateMktDepthL2_ValidInput() {
        int tickerId = 123;
        int position = 0;
        String marketMaker = "MM1";
        int operation = 1;
        int side = 2;
        double price = 10.50;
        int size = 100;
        String expectedOutput = "updateMktDepth: 123 0 MM1 1 2 10.5 100";
        String actualOutput = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateMktDepthL2_ZeroValues() {
        int tickerId = 0;
        int position = 0;
        String marketMaker = "";
        int operation = 0;
        int side = 0;
        double price = 0.0;
        int size = 0;
        String expectedOutput = "updateMktDepth: 0 0  0 0 0 0.0 0";
        String actualOutput = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateMktDepthL2_NegativeValues() {
        int tickerId = -1;
        int position = -5;
        String marketMaker = "MM2";
        int operation = -2;
        int side = -1;
        double price = -20.75;
        int size = -50;
        String expectedOutput = "updateMktDepth: -1 -5 MM2 -2 -1 -20.75 -50";
        String actualOutput = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateMktDepthL2_LargeValues() {
        int tickerId = Integer.MAX_VALUE;
        int position = Integer.MAX_VALUE;
        String marketMaker = "MM_MAX";
        int operation = Integer.MAX_VALUE;
        int side = Integer.MAX_VALUE;
        double price = Double.MAX_VALUE;
        int size = Integer.MAX_VALUE;
        String expectedOutput = "updateMktDepth: 2147483647 2147483647 MM_MAX 2147483647 2147483647 1.7976931348623157E308 2147483647";
        String actualOutput = EWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        assertEquals(expectedOutput, actualOutput);
    }
}
