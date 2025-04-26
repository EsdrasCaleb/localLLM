package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingBelow_34_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testSpRatingBelow() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedSpRatingBelow = "BBB-";
        // Act
        scannerSubscription.spRatingBelow(expectedSpRatingBelow);
        // Reflect to get the private field value
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingBelow");
        field.setAccessible(true);
        String actualSpRatingBelow = (String) field.get(scannerSubscription);
        // Assert
        assertEquals(expectedSpRatingBelow, actualSpRatingBelow);
    }
}
