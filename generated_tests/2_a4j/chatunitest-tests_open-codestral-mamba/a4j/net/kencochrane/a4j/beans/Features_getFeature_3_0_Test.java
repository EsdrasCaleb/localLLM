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

    private Features features;

    @BeforeEach
    public void setUp() {
        features = Mockito.mock(Features.class);
    }

    @Test
    public void testGetFeatureInBounds() {
        ArrayList<String> featureList = new ArrayList<>();
        featureList.add("Feature1");
        featureList.add("Feature2");
        featureList.add("Feature3");
        Mockito.when(features.getFeaturesArray()).thenReturn(featureList);
        String result = features.getFeature(1);
        assertEquals("Feature2", result);
    }

    @Test
    public void testGetFeatureOutOfBounds() {
        ArrayList<String> featureList = new ArrayList<>();
        featureList.add("Feature1");
        featureList.add("Feature2");
        featureList.add("Feature3");
        Mockito.when(features.getFeaturesArray()).thenReturn(featureList);
        String result = features.getFeature(3);
        assertNull(result);
    }
}
