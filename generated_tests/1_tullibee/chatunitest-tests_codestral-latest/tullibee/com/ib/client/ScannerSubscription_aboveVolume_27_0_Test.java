package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_27_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAboveVolume() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        int expectedVolume = 100;
        // Act
        scannerSubscription.aboveVolume(expectedVolume);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_aboveVolume");
        field.setAccessible(true);
        int actualVolume = (int) field.get(scannerSubscription);
        assertEquals(expectedVolume, actualVolume);
    }
}
