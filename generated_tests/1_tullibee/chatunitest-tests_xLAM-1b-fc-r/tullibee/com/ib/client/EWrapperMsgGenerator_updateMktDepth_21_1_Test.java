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

public class EWrapperMsgGenerator_updateMktDepth_21_1_Test {

    @Test
    public void updateMktDepthTest() {
        // Arrange
        int tickerId = 123;
        int position = 456;
        int operation = 789;
        int side = 1011;
        double price = 123.45;
        int size = 14;
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Act
        String result = eWrapperMsgGenerator.updateMktDepth(tickerId, position, operation, side, price, size);
        // Assert
        String expected = "SCANNER_PARAMETERS: " + tickerId + "," + position + "," + operation + "," + side + "," + price + "," + size;
        assertEquals(expected, result);
    }
}
