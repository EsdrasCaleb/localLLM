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

public class EWrapperMsgGenerator_updateMktDepth_21_0_Test {

    @Test
    public void testUpdateMktDepth() throws Exception {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int tickerId = 1;
        int position = 0;
        // Assuming 1 represents an add operation
        int operation = 1;
        // Assuming 0 represents buy
        int side = 0;
        double price = 100.5;
        int size = 10;
        // Act
        String result = invokeUpdateMktDepth(generator, tickerId, position, operation, side, price, size);
        // Assert
        String expected = "updateMktDepth: " + tickerId + " " + position + " " + operation + " " + side + " " + price + " " + size;
        assertEquals(expected, result);
    }

    private String invokeUpdateMktDepth(EWrapperMsgGenerator generator, int tickerId, int position, int operation, int side, double price, int size) throws Exception {
        // Use reflection to access the private method
        var method = EWrapperMsgGenerator.class.getDeclaredMethod("updateMktDepth", int.class, int.class, int.class, int.class, double.class, int.class);
        method.setAccessible(true);
        return (String) method.invoke(generator, tickerId, position, operation, side, price, size);
    }
}
