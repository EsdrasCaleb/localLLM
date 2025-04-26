package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAbovePriceSetsCorrectValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testPrice = 150.75;
        // When
        scannerSubscription.abovePrice(testPrice);
        // Then
        Field abovePriceField = ScannerSubscription.class.getDeclaredField("m_abovePrice");
        abovePriceField.setAccessible(true);
        double actualPrice = (double) abovePriceField.get(scannerSubscription);
        assertEquals(testPrice, actualPrice, "The abovePrice should be set to the provided value");
    }

    @Test
    public void testAbovePriceSetsMaxValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testPrice = Double.MAX_VALUE;
        // When
        scannerSubscription.abovePrice(testPrice);
        // Then
        Field abovePriceField = ScannerSubscription.class.getDeclaredField("m_abovePrice");
        abovePriceField.setAccessible(true);
        double actualPrice = (double) abovePriceField.get(scannerSubscription);
        assertEquals(testPrice, actualPrice, "The abovePrice should be set to Double.MAX_VALUE");
    }

    @Test
    public void testAbovePriceSetsMinValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testPrice = Double.MIN_VALUE;
        // When
        scannerSubscription.abovePrice(testPrice);
        // Then
        Field abovePriceField = ScannerSubscription.class.getDeclaredField("m_abovePrice");
        abovePriceField.setAccessible(true);
        double actualPrice = (double) abovePriceField.get(scannerSubscription);
        assertEquals(testPrice, actualPrice, "The abovePrice should be set to Double.MIN_VALUE");
    }

    @Test
    public void testAbovePriceSetsNegativeValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testPrice = -100.0;
        // When
        scannerSubscription.abovePrice(testPrice);
        // Then
        Field abovePriceField = ScannerSubscription.class.getDeclaredField("m_abovePrice");
        abovePriceField.setAccessible(true);
        double actualPrice = (double) abovePriceField.get(scannerSubscription);
        assertEquals(testPrice, actualPrice, "The abovePrice should be set to a negative value");
    }

    @Test
    public void testAbovePriceSetsZeroValue() throws NoSuchFieldException, IllegalAccessException {
        // Given
        double testPrice = 0.0;
        // When
        scannerSubscription.abovePrice(testPrice);
        // Then
        Field abovePriceField = ScannerSubscription.class.getDeclaredField("m_abovePrice");
        abovePriceField.setAccessible(true);
        double actualPrice = (double) abovePriceField.get(scannerSubscription);
        assertEquals(testPrice, actualPrice, "The abovePrice should be set to zero");
    }
}
