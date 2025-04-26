package net.kencochrane.a4j.util;

import java.lang.reflect.Field;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;

public class LoadProperties_instance_0_1_Test {

    @Test
    void testInstance_firstCallCreatesInstance() {
        LoadProperties instance1 = LoadProperties.instance();
        assertNotNull(instance1);
    }

    @Test
    void testInstance_subsequentCallsReturnSameInstance() {
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        assertSame(instance1, instance2);
    }

    @Test
    void testInstance_accessPrivateField() throws NoSuchFieldException, IllegalAccessException {
        LoadProperties instance = LoadProperties.instance();
        Field instanceField = LoadProperties.class.getDeclaredField("_instance");
        instanceField.setAccessible(true);
        LoadProperties retrievedInstance = (LoadProperties) instanceField.get(null);
        assertSame(instance, retrievedInstance);
    }

    @Test
    void testGetProperties_returnsPropertiesObject() {
        LoadProperties instance = LoadProperties.instance();
        Properties props = instance.getProperties();
        assertNotNull(props);
        assertTrue(props instanceof Properties);
    }

    @Test
    void testSetDefaultProperties_setsDefaultValues() throws NoSuchMethodException, IllegalAccessException, java.lang.reflect.InvocationTargetException {
        LoadProperties instance = LoadProperties.instance();
        Properties props = instance.getProperties();
        java.lang.reflect.Method method = LoadProperties.class.getDeclaredMethod("setDefaultProperties", Properties.class);
        method.setAccessible(true);
        method.invoke(instance, props);
        assertEquals("popcornmonste2-20", props.getProperty("associateID"));
        assertEquals("86400000", props.getProperty("cacheLife"));
        assertEquals("-", props.getProperty("URLSeperator"));
        assertEquals("/", props.getProperty("cacheDir"));
        assertEquals("http://xml.amazon.net/onca/xml3", props.getProperty("amazonServerURL"));
    }
}
