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
    void testUpdateMktDepthL2() {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int tickerId = 12345;
        int position = 0;
        String marketMaker = "MarketMaker1";
        int operation = 1;
        int side = 1;
        double price = 100.0;
        int size = 100;
        // Act
        String result = eWrapperMsgGenerator.updateMktDepthL2(tickerId, position, marketMaker, operation, side, price, size);
        // Assert
        assertEquals("updateMktDepth: 12345 0 buy MarketMaker1 1 100.0 100", result);
    }
}
