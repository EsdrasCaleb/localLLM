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

public class EWrapperMsgGenerator_updateMktDepthL2_22_0_Test {

    @Test
    public void testUpdateMktDepthL2() {
        // Test case 1: Basic test with positive values
        String result1 = EWrapperMsgGenerator.updateMktDepthL2(123, 456, "MarketMaker1", 1, 0, 123.45, 678);
        assertEquals("updateMktDepth: 123 456 MarketMaker1 1 0 123.45 678", result1);
        // Test case 2: Test with zero values
        String result2 = EWrapperMsgGenerator.updateMktDepthL2(0, 0, "", 0, 0, 0.0, 0);
        assertEquals("updateMktDepth: 0 0  0 0 0.0 0", result2);
        // Test case 3: Test with negative values
        String result3 = EWrapperMsgGenerator.updateMktDepthL2(-123, -456, "MarketMaker2", -1, -1, -123.45, -678);
        assertEquals("updateMktDepth: -123 -456 MarketMaker2 -1 -1 -123.45 -678", result3);
        // Test case 4: Test with large values
        String result4 = EWrapperMsgGenerator.updateMktDepthL2(Integer.MAX_VALUE, Integer.MAX_VALUE, "MarketMaker3", Integer.MAX_VALUE, Integer.MAX_VALUE, Double.MAX_VALUE, Integer.MAX_VALUE);
        assertEquals("updateMktDepth: 2147483647 2147483647 MarketMaker3 2147483647 2147483647 1.7976931348623157E308 2147483647", result4);
        // Test case 5: Test with special characters in marketMaker
        String result5 = EWrapperMsgGenerator.updateMktDepthL2(123, 456, "Ma:rket,Maker!@", 1, 0, 123.45, 678);
        assertEquals("updateMktDepth: 123 456 Ma:rket,Maker!@ 1 0 123.45 678", result5);
    }
}
