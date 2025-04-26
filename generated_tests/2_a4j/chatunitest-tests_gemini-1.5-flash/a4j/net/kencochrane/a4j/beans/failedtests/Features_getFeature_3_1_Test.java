package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_getFeature_3_1_Test {

    private Features features;

    @BeforeEach
    void setUp() {
        features = new Features();
    }

    @Test
    void testGetFeatureWithinBounds() {
        String[] featureArray = { "feature1", "feature2", "feature3" };
        features.setFeature(featureArray);
        assertEquals("feature1", features.getFeature(0));
        assertEquals("feature2", features.getFeature(1));
        assertEquals("feature3", features.getFeature(2));
    }

    @Test
    void testGetFeatureOutOfBounds() {
        String[] featureArray = { "feature1", "feature2", "feature3" };
        features.setFeature(featureArray);
        assertNull(features.getFeature(3));
        assertNull(features.getFeature(-1));
    }

    @Test
    void testGetFeatureEmptyList() {
        assertNull(features.getFeature(0));
    }

    @Test
    void testGetFeatureNullList() {
        // This test case is not possible to be performed without reflection, because the ArrayList is private
        try {
            Field featuresField = Features.class.getDeclaredField("features");
            featuresField.setAccessible(true);
            featuresField.set(features, null);
            assertThrows(NullPointerException.class, () -> features.getFeature(0));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }
}
