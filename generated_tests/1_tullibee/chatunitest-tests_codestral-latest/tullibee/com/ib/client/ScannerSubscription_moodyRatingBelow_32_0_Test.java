package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_32_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedRating = "A1";
        // Act
        scannerSubscription.moodyRatingBelow(expectedRating);
        // Reflect to get the private field value
        Field field = ScannerSubscription.class.getDeclaredField("m_moodyRatingBelow");
        field.setAccessible(true);
        String actualRating = (String) field.get(scannerSubscription);
        // Assert
        assertNotNull(actualRating);
        assertEquals(expectedRating, actualRating);
    }
}
