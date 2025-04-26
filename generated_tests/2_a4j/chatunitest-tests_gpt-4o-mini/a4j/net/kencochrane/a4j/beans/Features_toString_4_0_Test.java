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

public class Features_toString_4_0_Test {

    private Features features;

    @BeforeEach
    public void setUp() {
        features = new Features();
    }

    @Test
    public void testToString_EmptyFeatures() {
        // Arrange
        setFeaturesField(new ArrayList<>());
        // Act
        String result = features.toString();
        // Assert
        assertEquals("Feature is null or size 0\n", result);
    }

    @Test
    public void testToString_NullFeatures() throws Exception {
        // Arrange
        setFeaturesField(null);
        // Act
        String result = features.toString();
        // Assert
        assertEquals("Feature is null or size 0\n", result);
    }

    @Test
    public void testToString_SingleFeature() {
        // Arrange
        ArrayList<String> featureList = new ArrayList<>();
        featureList.add("Feature1");
        setFeaturesField(featureList);
        // Act
        String result = features.toString();
        // Assert
        assertEquals("# of Feature = 1\nFeature - Feature1\n", result);
    }

    @Test
    public void testToString_MultipleFeatures() {
        // Arrange
        ArrayList<String> featureList = new ArrayList<>();
        featureList.add("Feature1");
        featureList.add("Feature2");
        setFeaturesField(featureList);
        // Act
        String result = features.toString();
        // Assert
        assertEquals("# of Feature = 2\nFeature - Feature1\nFeature - Feature2\n", result);
    }

    private void setFeaturesField(ArrayList<String> featuresList) {
        try {
            Field featuresField = Features.class.getDeclaredField("features");
            featuresField.setAccessible(true);
            featuresField.set(features, featuresList);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
