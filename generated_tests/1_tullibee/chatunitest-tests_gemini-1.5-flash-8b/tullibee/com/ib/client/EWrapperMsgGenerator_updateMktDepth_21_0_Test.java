package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.lang.reflect.InvocationTargetException;
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

class EWrapperMsgGenerator_updateMktDepth_21_0_Test {

    @ParameterizedTest
    @CsvSource({ "123,0,BUY,10.50,100,123 0 BUY 10.5 100", "456,1,SELL,20.25,50,456 1 SELL 20.25 50", "789,-1,BUY,15.75,200,789 -1 BUY 15.75 200", "101,0,SELL,100.00,1,101 0 SELL 100.0 1" })
    void testUpdateMktDepth(int tickerId, int position, String side, double price, int size, String expectedOutput) {
        try {
            Method updateMktDepthMethod = EWrapperMsgGenerator.class.getDeclaredMethod("updateMktDepth", int.class, int.class, String.class, double.class, int.class);
            updateMktDepthMethod.setAccessible(true);
            String actualOutput = (String) updateMktDepthMethod.invoke(null, tickerId, position, side, price, size);
            assertEquals(expectedOutput, actualOutput);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            fail("Error invoking method: " + e.getMessage());
        }
    }

    @Test
    void testUpdateMktDepth_NullSide() {
        try {
            Method updateMktDepthMethod = EWrapperMsgGenerator.class.getDeclaredMethod("updateMktDepth", int.class, int.class, String.class, double.class, int.class);
            updateMktDepthMethod.setAccessible(true);
            String actualOutput = (String) updateMktDepthMethod.invoke(null, 123, 0, null, 10.50, 100);
            // Or assert that it throws an exception, depending on the expected behavior.
            assertNull(actualOutput);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            fail("Error invoking method: " + e.getMessage());
        }
    }
}
