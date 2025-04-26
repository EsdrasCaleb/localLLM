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

public class EWrapperMsgGenerator_updateMktDepth_21_0_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    public void testUpdateMktDepth() throws Exception {
        // Prepare test data
        int tickerId = 12345;
        int position = 1;
        // Insert
        int operation = 0;
        // Bid
        int side = 0;
        double price = 150.75;
        int size = 100;
        // Expected result
        String expectedResult = "Market Depth Update - Ticker ID: 12345, Position: 1, Operation: Insert, Side: Bid, Price: 150.75, Size: 100";
        // Use reflection to invoke the private method
        Method updateMktDepthMethod = EWrapperMsgGenerator.class.getDeclaredMethod("updateMktDepth", int.class, int.class, int.class, int.class, double.class, int.class);
        updateMktDepthMethod.setAccessible(true);
        // Invoke the method
        String result = (String) updateMktDepthMethod.invoke(eWrapperMsgGenerator, tickerId, position, operation, side, price, size);
        // Verify the result
        assertEquals(expectedResult, result);
        // Test other branches for operation and side
        // Operation: Update (1), Side: Ask (1)
        operation = 1;
        side = 1;
        expectedResult = "Market Depth Update - Ticker ID: 12345, Position: 1, Operation: Update, Side: Ask, Price: 150.75, Size: 100";
        result = (String) updateMktDepthMethod.invoke(eWrapperMsgGenerator, tickerId, position, operation, side, price, size);
        assertEquals(expectedResult, result);
        // Operation: Delete (2), Side: Bid (0)
        operation = 2;
        side = 0;
        expectedResult = "Market Depth Update - Ticker ID: 12345, Position: 1, Operation: Delete, Side: Bid, Price: 150.75, Size: 100";
        result = (String) updateMktDepthMethod.invoke(eWrapperMsgGenerator, tickerId, position, operation, side, price, size);
        assertEquals(expectedResult, result);
    }
}
