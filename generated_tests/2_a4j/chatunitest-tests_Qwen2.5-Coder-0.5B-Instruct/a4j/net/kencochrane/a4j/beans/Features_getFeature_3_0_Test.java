package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Features_getFeature_3_0_Test {

    private Features features;

    @BeforeEach
    public void setUp() {
        features = new Features();
    }

    @Test
    public void testGetFeature() {
        // Arrange
        features.setFeature(new String[] { "feature1", "feature2", "feature3" });
        int index = 1;
        // Act
        String result = features.getFeature(index);
        // Assert
        assertEquals("feature2", result);
    }
}
