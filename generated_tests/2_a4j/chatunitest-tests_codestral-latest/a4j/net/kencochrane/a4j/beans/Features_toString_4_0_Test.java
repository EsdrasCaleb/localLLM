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

class Features_toString_4_0_Test {

    @InjectMocks
    private Features features;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        features = new Features();
    }

    @Test
    void testToStringFeaturesNull() throws Exception {
        setFeaturesField(null);
        String result = features.toString();
        assertEquals("Feature is null or size 0\n", result);
    }

    @Test
    void testToStringFeaturesEmpty() throws Exception {
        setFeaturesField(new ArrayList<>());
        String result = features.toString();
        assertEquals("Feature is null or size 0\n", result);
    }

    @Test
    void testToStringFeaturesNotEmpty() throws Exception {
        ArrayList<String> featureList = new ArrayList<>();
        featureList.add("Feature1");
        featureList.add("Feature2");
        setFeaturesField(featureList);
        String result = features.toString();
        String expected = "# of Feature = 2\nFeature - Feature1\nFeature - Feature2\n";
        assertEquals(expected, result);
    }

    private void setFeaturesField(ArrayList<String> featureList) throws Exception {
        Field field = Features.class.getDeclaredField("features");
        field.setAccessible(true);
        field.set(features, featureList);
    }
}
