package com.densebrain.rif.client;

import java.lang.reflect.Field;
import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.RIFServiceStub;

public class RIFManager_getInvoker_0_0_Test {

    @Test
    void testGetInvokerExistingInvoker() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        RIFManager manager = new RIFManager("testUrl");
        RIFInvoker mockInvoker = Mockito.mock(RIFInvoker.class);
        Field invokerMapField = manager.getClass().getDeclaredField("invokerMap");
        invokerMapField.setAccessible(true);
        ((Hashtable<Class<?>, RIFInvoker>) invokerMapField.get(manager)).put(String.class, mockInvoker);
        RIFInvoker result = manager.getInvoker(String.class);
        assertSame(mockInvoker, result);
        verify(mockInvoker, never()).invoke(any());
    }

    @Test
    void testGetInvokerNewInvoker() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        RIFManager manager = new RIFManager("testUrl");
        Field invokerMapField = manager.getClass().getDeclaredField("invokerMap");
        invokerMapField.setAccessible(true);
        Hashtable<Class<?>, RIFInvoker> map = (Hashtable<Class<?>, RIFInvoker>) invokerMapField.get(manager);
        assertEquals(0, map.size());
        RIFInvoker result = manager.getInvoker(Integer.class);
        assertNotNull(result);
        assertEquals(1, map.size());
        assertTrue(map.containsKey(Integer.class));
        assertSame(result, map.get(Integer.class));
    }

    @Test
    void testGetInvokerConcurrentAccess() throws RemoteException, NoSuchFieldException, IllegalAccessException, InterruptedException {
        RIFManager manager = new RIFManager("testUrl");
        Field invokerMapField = manager.getClass().getDeclaredField("invokerMap");
        invokerMapField.setAccessible(true);
        Hashtable<Class<?>, RIFInvoker> map = (Hashtable<Class<?>, RIFInvoker>) invokerMapField.get(manager);
        assertEquals(0, map.size());
        Thread thread1 = new Thread(() -> {
            try {
                manager.getInvoker(Double.class);
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        });
        Thread thread2 = new Thread(() -> {
            try {
                manager.getInvoker(Double.class);
            } catch (RemoteException e) {
                e.printStackTrace();
            }
        });
        thread1.start();
        thread2.start();
        thread1.join();
        thread2.join();
        assertEquals(1, map.size());
        assertTrue(map.containsKey(Double.class));
    }

    static class RIFInvoker {

        public <T> T invoke(Object o) {
            return null;
        }
    }

    static class RIFManager {

        private Hashtable<Class<?>, RIFInvoker> invokerMap = new Hashtable<>();

        public RIFManager(String url) {
        }

        public RIFInvoker getInvoker(Class<?> clazz) throws RemoteException {
            RIFInvoker invoker = invokerMap.get(clazz);
            if (invoker == null) {
                invoker = new RIFInvoker();
                invokerMap.put(clazz, invoker);
            }
            return invoker;
        }
    }
}
