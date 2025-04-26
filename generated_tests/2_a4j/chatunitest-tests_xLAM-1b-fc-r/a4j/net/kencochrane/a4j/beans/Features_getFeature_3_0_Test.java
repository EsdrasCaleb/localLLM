package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_getFeature_3_0_Test {

    @Mock
    Features features;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetFeature() {
        // Given
        String[] testString = { "test1", "test2", "test3" };
        when(features.getFeature()).thenReturn(testString);
        // When
        String result = features.getFeature(0);
        // Then
        assertEquals("test1", result);
    }
}
