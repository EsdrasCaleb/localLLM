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

public class EWrapperMsgGenerator_tickPrice_0_2_Test {

    @Test
    public void testTickPrice() throws Exception {
        // Create an instance of the EWrapperMsgGenerator class
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Get the method to invoke
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("tickPrice", int.class, int.class, double.class, int.class);
        method.setAccessible(true);
        // Invoke the method and get the result
        String result = (String) method.invoke(eWrapperMsgGenerator, 1, 100, 123.45, 1);
        // Expected result
        String expectedResult = "The tickPrice method in the EWrapperMsgGenerator class is used to generate messages for tick prices. " + "The method accepts four parameters: " + "tickerId (an integer representing the unique identifier for the ticker), " + "field (an integer representing the field of the tick price), " + "price (a double representing the price of the tick), " + "and canAutoExecute (an integer representing a boolean value indicating whether the tick can be automatically executed). " + "The method returns a string that represents the tick price message with the tickerId, field, price, and canAutoExecute information. " + "The message format is: 'id=tickerId  field=fieldName=price canAutoExecute' or 'id=tickerId  field=fieldName=price noAutoExecute'.";
        assertEquals(expectedResult, result);
    }
}
