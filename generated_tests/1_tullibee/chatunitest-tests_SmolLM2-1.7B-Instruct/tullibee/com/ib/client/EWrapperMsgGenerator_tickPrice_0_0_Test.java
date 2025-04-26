package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Mock
    private EWrapperMsgGenerator wrapperMsgGenerator;

    @InjectMocks
    private EWrapperMsgGenerator underTest;

    @Test
    public void testTickPrice_ValidInputs_ReturnsExpectedString() {
        // Arrange
        int tickerId = 1;
        int field = 2;
        double price = 3.0;
        int canAutoExecute = 1;
        // Act
        String result = underTest.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals("id=1  field=2=3.0  canAutoExecute=1", result);
    }

    @Test
    public void testTickPrice_InvalidTickerId_ThrowsException() {
        // Arrange
        int tickerId = -1;
        int field = 2;
        double price = 3.0;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> underTest.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_InvalidField_ThrowsException() {
        // Arrange
        int tickerId = 1;
        int field = -1;
        double price = 3.0;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> underTest.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_InvalidPrice_ThrowsException() {
        // Arrange
        int tickerId = 1;
        int field = 2;
        double price = Double.NaN;
        int canAutoExecute = 1;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> underTest.tickPrice(tickerId, field, price, canAutoExecute));
    }

    @Test
    public void testTickPrice_InvalidCanAutoExecute_ThrowsException() {
        // Arrange
        int tickerId = 1;
        int field = 2;
        double price = 3.0;
        int canAutoExecute = -1;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> underTest.tickPrice(tickerId, field, price, canAutoExecute));
    }
}
