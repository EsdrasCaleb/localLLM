package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_31_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingAbove() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String rating = "A1";
        // Act
        scannerSubscription.moodyRatingAbove(rating);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertEquals(rating, field.get(scannerSubscription));
    }

    @Test
    public void testMoodyRatingAboveNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String rating = null;
        // Act
        scannerSubscription.moodyRatingAbove(rating);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        field.setAccessible(true);
        assertNull(field.get(scannerSubscription));
    }
}
