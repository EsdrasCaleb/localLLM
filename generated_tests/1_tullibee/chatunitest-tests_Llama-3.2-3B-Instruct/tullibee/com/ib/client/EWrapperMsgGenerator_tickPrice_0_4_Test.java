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

public class EWrapperMsgGenerator_tickPrice_0_4_Test {

    @Test
    public void testTickPrice() {
        // Arrange
        int tickerId = 12345;
        int field = 1;
        double price = 10.5;
        int canAutoExecute = 1;
        String expectedMessage = "id=" + tickerId + "  LastPrice=" + price + " canAutoExecute=" + canAutoExecute;
        // Act
        String actualMessage = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    public void testTickPrice_AutoExecuteFalse() {
        // Arrange
        int tickerId = 12345;
        int field = 1;
        double price = 10.5;
        int canAutoExecute = 0;
        String expectedMessage = "id=" + tickerId + "  LastPrice=" + price + " noAutoExecute";
        // Act
        String actualMessage = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals(expectedMessage, actualMessage);
    }

    @Test
    public void testTickPrice_NullTickerId() {
        // Arrange
        int tickerId = 0;
        int field = 1;
        double price = 10.5;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_NullField() {
        // Arrange
        int tickerId = 12345;
        int field = 0;
        double price = 10.5;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_NullPrice() {
        // Arrange
        int tickerId = 12345;
        int field = 1;
        double price = 0;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_NullCanAutoExecute() {
        // Arrange
        int tickerId = 12345;
        int field = 1;
        double price = 10.5;
        int canAutoExecute = 0;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute));
    }
}
