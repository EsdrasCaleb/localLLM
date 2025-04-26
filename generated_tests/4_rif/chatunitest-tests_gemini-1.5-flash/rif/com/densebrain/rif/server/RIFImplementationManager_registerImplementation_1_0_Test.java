package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Hashtable;
import java.util.Map;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.rmi.RemoteException;

public class RIFImplementationManager_registerImplementation_1_0_Test {

    @Test
    void testRegisterImplementationNullInterface() {
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        assertThrows(NullPointerException.class, () -> manager.registerImplementation(null, new Object()));
    }

    @Test
    void testRegisterImplementationNullImplementation() {
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        assertThrows(NullPointerException.class, () -> manager.registerImplementation(String.class, null));
    }

    @Test
    void testRegisterImplementationValid() throws NoSuchFieldException, IllegalAccessException {
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        manager.registerImplementation(String.class, "test");
        Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
        implementationMapField.setAccessible(true);
        Hashtable<String, Object> implementationMap = (Hashtable<String, Object>) implementationMapField.get(manager);
        assertEquals("test", implementationMap.get(String.class.getName()));
    }

    @Test
    void testRegisterImplementationOverwrite() throws NoSuchFieldException, IllegalAccessException {
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        manager.registerImplementation(String.class, "test1");
        manager.registerImplementation(String.class, "test2");
        Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
        implementationMapField.setAccessible(true);
        Hashtable<String, Object> implementationMap = (Hashtable<String, Object>) implementationMapField.get(manager);
        assertEquals("test2", implementationMap.get(String.class.getName()));
    }

    interface TestInterface {
    }

    class TestImplementation implements TestInterface {
    }

    @Test
    void testRegisterImplementationInterface() throws NoSuchFieldException, IllegalAccessException {
        RIFImplementationManager manager = RIFImplementationManager.getInstance();
        TestImplementation impl = new TestImplementation();
        manager.registerImplementation(TestInterface.class, impl);
        Field implementationMapField = RIFImplementationManager.class.getDeclaredField("implementationMap");
        implementationMapField.setAccessible(true);
        Hashtable<String, Object> implementationMap = (Hashtable<String, Object>) implementationMapField.get(manager);
        assertSame(impl, implementationMap.get(TestInterface.class.getName()));
    }
}
