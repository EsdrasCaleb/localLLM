package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_toString_4_0_Test {

    @Test
    void testToString_nullFeatures() {
        Features features = new Features();
        // Use reflection to set features to null
        try {
            Field featuresField = Features.class.getDeclaredField("features");
            featuresField.setAccessible(true);
            featuresField.set(features, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access features field: " + e.getMessage());
        }
        assertEquals("Feature is null or size 0\n", features.toString());
    }

    @Test
    void testToString_emptyFeatures() {
        Features features = new Features();
        // Use reflection to set features to an empty ArrayList
        try {
            Field featuresField = Features.class.getDeclaredField("features");
            featuresField.setAccessible(true);
            featuresField.set(features, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access features field: " + e.getMessage());
        }
        assertEquals("Feature is null or size 0\n", features.toString());
    }

    @Test
    void testToString_nonEmptyFeatures() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature1", "Feature2", "Feature3" });
        assertEquals("# of Feature = 3\n" + "Feature - Feature1\n" + "Feature - Feature2\n" + "Feature - Feature3\n", features.toString());
    }

    @Test
    void testToString_FeaturesWithNull() {
        Features features = new Features();
        String[] featureArray = { "Feature1", null, "Feature3" };
        features.setFeature(featureArray);
        assertEquals("# of Feature = 3\n" + "Feature - Feature1\n" + "Feature - null\n" + "Feature - Feature3\n", features.toString());
    }

    @Test
    void testToString_singleFeature() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature1" });
        assertEquals("# of Feature = 1\n" + "Feature - Feature1\n", features.toString());
    }
}
