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

    @InjectMocks
    private Features features;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString_FeaturesNotEmpty() throws Exception {
        // Arrange
        ArrayList<String> mockFeatures = new ArrayList<>();
        mockFeatures.add("Feature1");
        mockFeatures.add("Feature2");
        Field featuresField = Features.class.getDeclaredField("features");
        featuresField.setAccessible(true);
        featuresField.set(features, mockFeatures);
        // Act
        String result = features.toString();
        // Assert
        String expected = "# of Feature = 2\nFeature - Feature1\nFeature - Feature2\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_FeaturesEmpty() throws Exception {
        // Arrange
        ArrayList<String> mockFeatures = new ArrayList<>();
        Field featuresField = Features.class.getDeclaredField("features");
        featuresField.setAccessible(true);
        featuresField.set(features, mockFeatures);
        // Act
        String result = features.toString();
        // Assert
        String expected = "Feature is null or size 0\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_FeaturesNull() throws Exception {
        // Arrange
        Field featuresField = Features.class.getDeclaredField("features");
        featuresField.setAccessible(true);
        featuresField.set(features, null);
        // Act
        String result = features.toString();
        // Assert
        String expected = "Feature is null or size 0\n";
        assertEquals(expected, result);
    }
}
