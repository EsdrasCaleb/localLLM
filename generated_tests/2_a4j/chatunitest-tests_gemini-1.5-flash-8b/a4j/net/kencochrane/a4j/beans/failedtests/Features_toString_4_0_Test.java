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
    void setUp() {
        features = new Features();
    }

    @Test
    void testToString_emptyFeatures() {
        assertEquals("Feature is null or size 0\n", features.toString());
    }

    @Test
    void testToString_nullFeatures() {
        features = new Features();
        features.setFeature(null);
        assertEquals("Feature is null or size 0\n", features.toString());
    }

    @Test
    void testToString_singleFeature() {
        String[] featuresArray = { "Feature1" };
        features.setFeature(featuresArray);
        assertEquals("# of Feature = 1\nFeature - Feature1\n", features.toString());
    }

    @Test
    void testToString_multipleFeatures() {
        String[] featuresArray = { "Feature1", "Feature2", "Feature3" };
        features.setFeature(featuresArray);
        assertEquals("# of Feature = 3\nFeature - Feature1\nFeature - Feature2\nFeature - Feature3\n", features.toString());
    }

    @Test
    void testToString_mixedFeatures() {
        String[] featuresArray = { "Feature1", null, "Feature3" };
        features.setFeature(featuresArray);
        assertEquals("# of Feature = 3\nFeature - Feature1\nFeature - null\nFeature - Feature3\n", features.toString());
    }
}
