package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Features_getFeature_3_0_Test {

    @InjectMocks
    private Features features;

    @Mock
    private ArrayList<String> mockFeatures;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        features.setFeature(new String[] { "Feature1", "Feature2", "Feature3" });
    }

    @Test
    void testGetFeatureWithinBounds() {
        String result = features.getFeature(1);
        assertEquals("Feature2", result);
    }

    @Test
    void testGetFeatureOutOfBounds() {
        String result = features.getFeature(5);
        assertNull(result);
    }

    @Test
    void testGetFeatureEmptyList() {
        features.setFeature(new String[] {});
        String result = features.getFeature(0);
        assertNull(result);
    }

    @Test
    void testGetFeatureNegativeIndex() {
        String result = features.getFeature(-1);
        assertNull(result);
    }

    @Test
    void testGetFeatureBoundaryIndex() {
        String result = features.getFeature(2);
        assertEquals("Feature3", result);
    }
}
