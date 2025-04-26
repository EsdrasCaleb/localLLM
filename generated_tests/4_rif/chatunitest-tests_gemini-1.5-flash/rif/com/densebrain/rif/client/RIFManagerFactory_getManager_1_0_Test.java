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

@ExtendWith(MockitoExtension.class)
class RIFManagerFactory_getManager_1_0_Test {

    @Test
    void testGetManagerNewManager() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        RIFManagerFactory factory = new RIFManagerFactory();
        Field managerMapField = RIFManagerFactory.class.getDeclaredField("managerMap");
        managerMapField.setAccessible(true);
        Hashtable<String, RIFManager> managerMap = (Hashtable<String, RIFManager>) managerMapField.get(factory);
        assertEquals(0, managerMap.size());
        RIFManager result = factory.getManager("testUrl");
        assertNotNull(result);
        assertEquals(1, managerMap.size());
        assertEquals(result, managerMap.get("testUrl"));
        assertEquals("testUrl/rif/services/RIFService", result.toString());
    }

    @Test
    void testGetManagerConcurrentNewManager() throws RemoteException, NoSuchFieldException, IllegalAccessException, InterruptedException {
        RIFManagerFactory factory = new RIFManagerFactory();
        Field managerMapField = RIFManagerFactory.class.getDeclaredField("managerMap");
        managerMapField.setAccessible(true);
        Hashtable<String, RIFManager> managerMap = (Hashtable<String, RIFManager>) managerMapField.get(factory);
        assertEquals(0, managerMap.size());
        Thread thread1 = new Thread(() -> {
            try {
                factory.getManager("testUrl");
            } catch (RemoteException e) {
                e.printStackTrace();
                fail("Unexpected RemoteException");
            }
        });
        Thread thread2 = new Thread(() -> {
            try {
                factory.getManager("testUrl");
            } catch (RemoteException e) {
                e.printStackTrace();
                fail("Unexpected RemoteException");
            }
        });
        thread1.start();
        thread2.start();
        thread1.join();
        thread2.join();
        assertEquals(1, managerMap.size());
    }

    static class RIFManager {

        private String url;

        RIFManager(String url) {
            this.url = url + "/rif/services/RIFService";
        }

        @Override
        public String toString() {
            return url;
        }
    }

    static class RIFManagerFactory {

        private Hashtable<String, RIFManager> managerMap = new Hashtable<>();

        public RIFManager getManager(String url) throws RemoteException {
            if (managerMap.containsKey(url)) {
                return managerMap.get(url);
            } else {
                RIFManager manager = new RIFManager(url);
                managerMap.put(url, manager);
                return manager;
            }
        }
    }
}
