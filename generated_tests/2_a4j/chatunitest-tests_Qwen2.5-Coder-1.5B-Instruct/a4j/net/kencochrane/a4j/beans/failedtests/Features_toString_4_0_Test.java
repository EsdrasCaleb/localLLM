package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Features_toString_4_0_Test {

    private Features features;

    @BeforeEach
    public void setUp() {
        features = new Features();
    }

    @Test
    public void testToStringWithFeatures() {
        // Arrange
        features.setFeature(new String[] { "feature1", "feature2" });
        // Act
        String result = features.toString();
        // Assert
        assertEquals("# of Feature = 2\nFeature - feature1\nFeature - feature2\n", result);
    }

    @Test
    public void testToStringWithoutFeatures() {
        // Arrange
        features.setFeature(null);
        // Act
        String result = features.toString();
        // Assert
        assertEquals("Feature is null or size 0", result);
    }

    @Test
    public void testToStringWithEmptyFeatures() {
        // Arrange
        features.setFeature(new String[] {});
        // Act
        String result = features.toString();
        // Assert
        assertEquals("Feature is null or size 0", result);
    }
}
